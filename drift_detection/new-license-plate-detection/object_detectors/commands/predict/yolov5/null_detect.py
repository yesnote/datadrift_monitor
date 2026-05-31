from commands.predict.common import *
from commands.utils.predict_utils import (
    _bbox_loss_xywh_tensor,
    _class_loss_tensor,
    _flatten_raw_prediction_layers,
    _objectness_loss_tensor,
    _plain_iou_xywh_tensor,
    _yolo_offset_loss_tensor,
)


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    bbox_loss = parsed["null_detect_bbox_loss"]
    cls_loss = parsed["null_detect_cls_loss"]
    obj_loss = parsed["null_detect_obj_loss"]
    bbox_direction = parsed["null_detect_bbox_direction"]
    cls_direction = parsed["null_detect_cls_direction"]
    obj_direction = parsed["null_detect_obj_direction"]
    feature_set = parsed["null_detect_feature_set"]

    if not save_csv:
        return

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _boxes_to_original_xyxy(boxes: torch.Tensor, ratio, pad, image_tensor: torch.Tensor):
        if boxes.numel() == 0:
            return boxes.clone()
        out = boxes.clone()
        ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
        pad_w, pad_h = float(pad[0]), float(pad[1])
        img_h = int(image_tensor.shape[-2])
        img_w = int(image_tensor.shape[-1])
        out[..., [0, 2]] = (out[..., [0, 2]] - pad_w) / max(ratio_w, 1e-12)
        out[..., [1, 3]] = (out[..., [1, 3]] - pad_h) / max(ratio_h, 1e-12)
        out[..., [0, 2]] = out[..., [0, 2]].clamp(0, img_w)
        out[..., [1, 3]] = out[..., [1, 3]].clamp(0, img_h)
        return out

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = [] if feature_set == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
    null_feature_names = (
        ["bbox_loss", "obj_loss", "cls_loss"]
        if feature_set == "losses_only"
        else ["final_score", "size", "circum", "size_circum", "bbox_loss", "obj_loss", "cls_loss"]
    )
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        *output_feature_names,
        *null_feature_names,
    ]
    output_csv = run_dir / "null_detect.csv"

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "feature_compute_sec"],
        device=device,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            t_detector = timing.start()
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                raw_layers = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 else None
                raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
                if raw_anchor_priors is None:
                    raise RuntimeError("null_detect requires YOLO anchor priors, but detector.model() did not return them.")
                raw_flat = _flatten_raw_prediction_layers(raw_layers)
                if raw_flat is None:
                    raise RuntimeError("null_detect requires raw YOLO prediction layers, but detector.model() did not return them.")
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=raw_prediction,
                    logits=raw_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )
            detector_inference_sec = timing.elapsed(t_detector)

            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_img = raw_prediction[sample_idx].float()
                raw_img = raw_flat[sample_idx]
                anchor_img = raw_anchor_priors[sample_idx]
                batch_items += int(det_b.shape[0])

                for pred_idx, box in enumerate(det_b):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    if raw_pred_idx >= int(pred_img.shape[0]) or raw_pred_idx >= int(raw_img.shape[0]) or raw_pred_idx >= int(anchor_img.shape[0]):
                        continue

                    t_feature = timing.start()
                    pred_row = pred_img[raw_pred_idx]
                    raw_row = raw_img[raw_pred_idx]
                    anchor_xywh = anchor_img[raw_pred_idx].to(dtype=pred_row.dtype, device=pred_row.device)
                    prob_values = {}
                    if feature_set != "losses_only":
                        pred_probs = pred_row[5:].detach().float() if pred_row.shape[0] > 5 else torch.zeros((0,), dtype=torch.float32, device=device)
                        prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)}
                        for prob_idx in range(max(0, num_classes)):
                            prob_values[f"prob_{prob_idx}"] = (
                                pred_probs[prob_idx]
                                if prob_idx < int(pred_probs.shape[0])
                                else torch.zeros((), dtype=torch.float32, device=device)
                            )

                    fbox_orig = _boxes_to_original_xyxy(box[:4].view(1, 4), ratios[sample_idx], pads[sample_idx], image_list[sample_idx]).view(4)
                    fx1_t, fy1_t, fx2_t, fy2_t = fbox_orig.unbind()
                    shape_values = {}
                    if feature_set != "losses_only":
                        width = torch.abs(fx2_t - fx1_t)
                        height = torch.abs(fy2_t - fy1_t)
                        size = width * height
                        circum = width + height
                        shape_values = {
                            "final_score": box[4],
                            "size": size,
                            "circum": circum,
                            "size_circum": size / circum.clamp(min=1e-12),
                        }

                    if str(bbox_loss).strip().lower().startswith("offset_"):
                        bbox_loss_value = _yolo_offset_loss_tensor(
                            raw_row[:4],
                            mode=bbox_loss,
                            reduction="mean",
                            direction=bbox_direction,
                        )
                    else:
                        bbox_loss_value = _bbox_loss_xywh_tensor(
                            pred_row[:4],
                            anchor_xywh,
                            mode=bbox_loss,
                            reduction="mean",
                            direction=bbox_direction,
                        )
                    target_iou = _plain_iou_xywh_tensor(pred_row[:4].detach(), anchor_xywh.detach()).to(
                        dtype=raw_row.dtype,
                        device=raw_row.device,
                    ).reshape(()).clamp(min=0.0, max=1.0)
                    obj_loss_value = _objectness_loss_tensor(
                        raw_row[4],
                        target_iou,
                        mode=obj_loss,
                        direction=obj_direction,
                        reduction="sum",
                    )
                    cls_logits = raw_row[5:]
                    if cls_logits.numel() == 0:
                        cls_loss_value = torch.zeros((), dtype=torch.float32, device=device)
                    else:
                        cls_target_value = (
                            0.5
                            if str(cls_loss).strip().lower() == "bcewithlogits"
                            else 1.0 / float(cls_logits.numel())
                        )
                        uniform_target = torch.full_like(cls_logits, cls_target_value)
                        cls_loss_value = _class_loss_tensor(
                            cls_logits,
                            uniform_target,
                            class_idx=None,
                            mode=cls_loss,
                            direction=cls_direction,
                            reduction="sum",
                        )
                    feature_compute_sec += timing.elapsed(t_feature)

                    cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                    if isinstance(detector.names, dict):
                        pred_class = detector.names.get(cls_idx, str(cls_idx))
                    elif isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
                        pred_class = detector.names[cls_idx]
                    else:
                        pred_class = str(cls_idx)

                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": _to_float(fx1_t),
                            "ymin": _to_float(fy1_t),
                            "xmax": _to_float(fx2_t),
                            "ymax": _to_float(fy2_t),
                            "score": float(box[4].detach().cpu().item()),
                            "pred_class": pred_class,
                            **{key: _to_float(value) for key, value in prob_values.items()},
                            **{key: _to_float(value) for key, value in shape_values.items()},
                            "bbox_loss": _to_float(bbox_loss_value),
                            "obj_loss": _to_float(obj_loss_value),
                            "cls_loss": _to_float(cls_loss_value),
                        }
                    )

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            output_file.flush()
            del infer_batch, model_output, raw_prediction, raw_logits, raw_flat, raw_anchor_priors, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
