from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms
from commands.predict.fcos.layer_grad import (
    _build_fcos_losses,
    _norm_direction,
    _norm_loss_name,
    _prediction_class_name,
)


def _parse_fcos_null_detect_config(config):
    out = config.get("output", {})
    active = out.get("null_detect", {}) if isinstance(out.get("null_detect", {}), dict) else {}
    feature_set = str(active.get("feature_set", "full")).strip().lower().replace("-", "_")
    if feature_set not in {"full", "losses_only"}:
        raise ValueError("Unsupported FCOS null_detect.feature_set. Supported values: full, losses_only.")

    bbox_loss = _norm_loss_name(
        active.get("bbox_loss", "l1"),
        "l1",
        {"l1", "l2"},
        aliases={"box_l1": "l1", "box_l2": "l2"},
    )
    cls_loss = _norm_loss_name(
        active.get("cls_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "kl"},
        aliases={"bce": "bcewithlogits"},
    )
    cnt_loss = _norm_loss_name(
        active.get("cnt_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "abs_diff", "signed_diff"},
        aliases={"bce": "bcewithlogits", "abs": "abs_diff", "signed": "signed_diff"},
    )
    bbox_direction = _norm_direction(active.get("bbox_direction", "pred_to_target"))
    cls_direction = _norm_direction(active.get("cls_direction", "pred_to_target"))
    cnt_direction = _norm_direction(active.get("cnt_direction", "pred_to_target"))
    if cls_direction == "target_to_pred" and cls_loss != "kl":
        raise ValueError("FCOS null_detect.cls_direction=target_to_pred is only supported when cls_loss=kl.")
    if cnt_direction == "target_to_pred" and cnt_loss != "signed_diff":
        raise ValueError("FCOS null_detect.cnt_direction=target_to_pred is only supported when cnt_loss=signed_diff.")

    return {
        "feature_set": feature_set,
        "bbox_loss": bbox_loss,
        "cls_loss": cls_loss,
        "cnt_loss": cnt_loss,
        "bbox_direction": bbox_direction,
        "cls_direction": cls_direction,
        "cnt_direction": cnt_direction,
    }


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _selected_prob_vector(selected_logits, raw_prediction, pred_idx, num_classes, device):
    if selected_logits is not None and pred_idx < int(selected_logits.shape[0]) and selected_logits.shape[1] > 0:
        probs = torch.sigmoid(selected_logits[pred_idx].detach().float())
    elif raw_prediction is not None and pred_idx < int(raw_prediction.shape[0]) and raw_prediction.shape[1] > 6:
        probs = raw_prediction[pred_idx, 6:].detach().float()
    else:
        probs = torch.zeros((0,), dtype=torch.float32, device=device)
    if probs.numel() < num_classes:
        padded = torch.zeros((num_classes,), dtype=torch.float32, device=device)
        if probs.numel():
            padded[: probs.numel()] = probs[:num_classes].to(device=device, dtype=torch.float32)
        return padded
    return probs[:num_classes].to(device=device, dtype=torch.float32)


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed_common = parse_output_config(config.get("output", {}))
    save_csv = parsed_common["save_csv_enabled"]
    unit = parsed_common["unit"]
    null_cfg = _parse_fcos_null_detect_config(config)

    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_fcos", False)):
        raise ValueError("commands.predict.fcos.null_detect requires model.type=fcos.")

    num_classes = len(detector.names) if detector.names is not None else 80
    output_feature_names = [] if null_cfg["feature_set"] == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
    null_feature_names = (
        ["bbox_loss", "cls_loss", "cnt_loss"]
        if null_cfg["feature_set"] == "losses_only"
        else ["final_score", "size", "circum", "size_circum", "bbox_loss", "cls_loss", "cnt_loss"]
    )
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
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
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            fcos_preprocessed = detector.preprocess_images(infer_batch)

            t_detector = timing.start()
            with torch.no_grad():
                model_output = detector.forward_layer_grad(fcos_preprocessed)
                selected_preds, selected_logits, _selected_objectness, selected_indices = select_fcos_post_nms(
                    detector,
                    model_output["post_prediction"],
                    model_output["post_logits"],
                    model_output["post_indices"],
                    conf_thres=float(getattr(detector, "confidence", getattr(detector, "conf_thresh", 0.05))),
                )
            detector_inference_sec = timing.elapsed(t_detector)

            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                logits = (
                    selected_logits[sample_idx]
                    if selected_logits and sample_idx < len(selected_logits)
                    else torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                )
                raw_keep = (
                    selected_indices[sample_idx]
                    if selected_indices and sample_idx < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                raw_prediction = (
                    model_output["post_prediction"][sample_idx]
                    if model_output.get("post_prediction") and sample_idx < len(model_output["post_prediction"])
                    else torch.zeros((0, 6 + num_classes), dtype=torch.float32, device=device)
                )
                batch_items += int(det.shape[0])

                for pred_idx in range(int(det.shape[0])):
                    raw_idx = int(raw_keep[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep.shape[0]) else pred_idx
                    final_box = det[pred_idx, :4]
                    final_cls = int(det[pred_idx, 5].detach().cpu().item()) if det.shape[1] > 5 else 0

                    t_feature = timing.start()
                    losses = _build_fcos_losses(
                        target_mode="null_target",
                        target_values=["bbox_loss", "cls_loss", "cnt_loss"],
                        model_output=model_output,
                        image_idx=sample_idx,
                        pred_idx=pred_idx,
                        final_box=final_box,
                        final_cls=final_cls,
                        raw_idx=raw_idx,
                        cand_score_threshold=0.0,
                        bbox_loss=null_cfg["bbox_loss"],
                        cls_loss=null_cfg["cls_loss"],
                        cnt_loss=null_cfg["cnt_loss"],
                        bbox_direction=null_cfg["bbox_direction"],
                        cls_direction=null_cfg["cls_direction"],
                        cnt_direction=null_cfg["cnt_direction"],
                        timing=timing,
                        timing_accumulator={"candidate_search_sec": 0.0, "loss_compute_sec": 0.0},
                    )
                    feature_values = {}
                    if null_cfg["feature_set"] != "losses_only":
                        probs = _selected_prob_vector(logits, raw_prediction, pred_idx, num_classes, device)
                        feature_values["prob_sum"] = probs.sum() if probs.numel() else torch.zeros((), dtype=torch.float32, device=device)
                        for prob_idx in range(max(0, num_classes)):
                            feature_values[f"prob_{prob_idx}"] = probs[prob_idx] if prob_idx < int(probs.shape[0]) else torch.zeros((), dtype=torch.float32, device=device)
                        width = torch.abs(final_box[2] - final_box[0])
                        height = torch.abs(final_box[3] - final_box[1])
                        size = width * height
                        circum = width + height
                        feature_values.update(
                            {
                                "final_score": det[pred_idx, 4],
                                "size": size,
                                "circum": circum,
                                "size_circum": size / circum.clamp(min=1e-12),
                            }
                        )
                    feature_values.update(
                        {
                            "bbox_loss": losses.get("bbox_loss", torch.zeros((), dtype=torch.float32, device=device)),
                            "cls_loss": losses.get("cls_loss", torch.zeros((), dtype=torch.float32, device=device)),
                            "cnt_loss": losses.get("cnt_loss", torch.zeros((), dtype=torch.float32, device=device)),
                        }
                    )
                    feature_compute_sec += timing.elapsed(t_feature)

                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_idx,
                            "xmin": _to_float(final_box[0]),
                            "ymin": _to_float(final_box[1]),
                            "xmax": _to_float(final_box[2]),
                            "ymax": _to_float(final_box[3]),
                            "score": _to_float(det[pred_idx, 4]),
                            "pred_class": _prediction_class_name(detector, final_cls),
                            **{key: _to_float(value) for key, value in feature_values.items()},
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
            if hasattr(detector, "_clear_last_pre_nms_predictions"):
                detector._clear_last_pre_nms_predictions()
            del infer_batch, fcos_preprocessed, model_output, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
