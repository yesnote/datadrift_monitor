from commands.predict.common import *
from commands.utils.predict_utils import (
    _concat_rpn_prediction_layers,
    _filter_rpn_proposals_with_indices,
    _resize_boxes_xyxy_tensor,
    build_faster_rcnn_null_losses_by_stage,
)


def _zero(device, dtype=torch.float32):
    return torch.zeros((), dtype=dtype, device=device)


def _loss_cfg(null_cfg, section, key, default):
    value = null_cfg.get(section, {})
    if isinstance(value, dict) and key in value:
        return value.get(key, default)
    return null_cfg.get(f"{section}_{key}", null_cfg.get(key, default))


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    null_cfg = config.get("output", {}).get("null_detect", {}) or {}
    rpn_bbox_loss = str(_loss_cfg(null_cfg, "rpn", "bbox_loss", "offset_l1")).strip().lower()
    rpn_obj_loss = str(_loss_cfg(null_cfg, "rpn", "obj_loss", parsed["null_detect_obj_loss"])).strip().lower()
    rpn_bbox_direction = str(_loss_cfg(null_cfg, "rpn", "bbox_direction", "pred_to_target")).strip().lower()
    rpn_obj_direction = str(_loss_cfg(null_cfg, "rpn", "obj_direction", parsed["null_detect_obj_direction"])).strip().lower()
    roi_bbox_loss = str(_loss_cfg(null_cfg, "roi", "bbox_loss", parsed["null_detect_bbox_loss"])).strip().lower()
    roi_cls_loss = str(_loss_cfg(null_cfg, "roi", "cls_loss", parsed["null_detect_cls_loss"])).strip().lower()
    roi_bbox_direction = str(_loss_cfg(null_cfg, "roi", "bbox_direction", parsed["null_detect_bbox_direction"])).strip().lower()
    roi_cls_direction = str(_loss_cfg(null_cfg, "roi", "cls_direction", parsed["null_detect_cls_direction"])).strip().lower()
    feature_set = parsed["null_detect_feature_set"]

    if not save_csv:
        return

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_faster_rcnn", False)):
        raise NotImplementedError("faster_rcnn null_detect runner requires a Faster R-CNN detector.")

    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = [] if feature_set == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
    null_feature_names = (
        ["rpn_bbox_loss", "rpn_obj_loss", "roi_bbox_loss", "roi_cls_loss"]
        if feature_set == "losses_only"
        else [
            "final_score", "size", "circum", "size_circum",
            "rpn_bbox_loss", "rpn_obj_loss", "roi_bbox_loss", "roi_cls_loss",
        ]
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

    model = detector.detector_model
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)

            was_training = model.training
            model.eval()
            with torch.no_grad():
                image_list = [
                    img.to(detector.device, non_blocking=True) if img.device != detector.device else img
                    for img in infer_batch
                ]
                original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
                transformed_images, _targets = model.transform(image_list, None)
                t_detector = timing.start()
                features = model.backbone(transformed_images.tensors)
                if isinstance(features, torch.Tensor):
                    features = {"0": features}
                features_list = list(features.values())
                rpn_objectness_list, rpn_bbox_delta_list = model.rpn.head(features_list)
                num_anchors_per_level = [
                    int(obj_per_level.shape[1] * obj_per_level.shape[2] * obj_per_level.shape[3])
                    for obj_per_level in rpn_objectness_list
                ]
                rpn_anchors = model.rpn.anchor_generator(transformed_images, features_list)
                rpn_objectness_flat, rpn_bbox_deltas_flat = _concat_rpn_prediction_layers(
                    rpn_objectness_list,
                    rpn_bbox_delta_list,
                )
                rpn_decoded_for_roi = model.rpn.box_coder.decode(rpn_bbox_deltas_flat.detach(), rpn_anchors).view(
                    len(image_list), -1, 4
                )
                proposals, _proposal_scores, proposal_to_rpn_raw_indices = _filter_rpn_proposals_with_indices(
                    model.rpn,
                    rpn_decoded_for_roi,
                    rpn_objectness_flat.detach(),
                    transformed_images.image_sizes,
                    num_anchors_per_level,
                )
                proposal_offsets = []
                running_proposal_offset = 0
                for proposal_img in proposals:
                    proposal_offsets.append(running_proposal_offset)
                    running_proposal_offset += int(proposal_img.shape[0])

                roi_heads = model.roi_heads
                box_features = roi_heads.box_roi_pool(features, proposals, transformed_images.image_sizes)
                box_features = roi_heads.box_head(box_features)
                class_logits, box_regression = roi_heads.box_predictor(box_features)
                detections = detector._pre_nms_detections_with_logits(
                    class_logits=class_logits,
                    box_regression=box_regression,
                    proposals=proposals,
                    image_shapes=transformed_images.image_sizes,
                )
                detections = model.transform.postprocess(detections, transformed_images.image_sizes, original_image_sizes)

                proposal_indices_by_img = []
                labels_internal_by_img = []
                for det_img in detections:
                    labels_internal_all = det_img["labels"].to(detector.device)
                    _labels_out, valid = detector._map_labels_to_output(labels_internal_all)
                    valid &= det_img["scores"].to(detector.device) > 0.0
                    proposal_indices_by_img.append(det_img["proposal_indices"].to(detector.device)[valid])
                    labels_internal_by_img.append(labels_internal_all[valid])

                raw_prediction, raw_logits = detector._detections_to_contract(
                    detections,
                    detector.device,
                    include_class_features=True,
                )
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
            if was_training:
                model.train()
            detector_inference_sec = timing.elapsed(t_detector)

            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_img = raw_prediction[sample_idx]
                logit_img = raw_logits[sample_idx] if raw_logits is not None else None
                proposals_img = _resize_boxes_xyxy_tensor(
                    proposals[sample_idx],
                    transformed_images.image_sizes[sample_idx],
                    original_image_sizes[sample_idx],
                )
                proposal_to_rpn_raw_idx_img = (
                    proposal_to_rpn_raw_indices[sample_idx]
                    if proposal_to_rpn_raw_indices and sample_idx < len(proposal_to_rpn_raw_indices)
                    else None
                )
                proposal_indices_img = proposal_indices_by_img[sample_idx] if sample_idx < len(proposal_indices_by_img) else None
                labels_internal_img = labels_internal_by_img[sample_idx] if sample_idx < len(labels_internal_by_img) else None
                proposal_offset_img = proposal_offsets[sample_idx] if sample_idx < len(proposal_offsets) else 0
                rpn_objectness_img = rpn_objectness_flat[sample_idx]
                rpn_bbox_deltas_img = rpn_bbox_deltas_flat[sample_idx]
                rpn_anchors_img = rpn_anchors[sample_idx]

                batch_items += int(det_b.shape[0])
                for pred_idx, box in enumerate(det_b):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    t_feature = timing.start()
                    prob_values = {}
                    if feature_set != "losses_only":
                        if 0 <= raw_pred_idx < int(pred_img.shape[0]) and pred_img.shape[1] > 6:
                            pred_probs = pred_img[raw_pred_idx, 6:].detach().float()
                        else:
                            pred_probs = torch.zeros((0,), dtype=torch.float32, device=device)
                        prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else _zero(device)}
                        for prob_idx in range(max(0, num_classes)):
                            prob_values[f"prob_{prob_idx}"] = (
                                pred_probs[prob_idx]
                                if prob_idx < int(pred_probs.shape[0])
                                else _zero(device)
                            )

                    shape_values = {}
                    if feature_set != "losses_only":
                        width = torch.abs(box[2] - box[0])
                        height = torch.abs(box[3] - box[1])
                        size = width * height
                        circum = width + height
                        shape_values = {
                            "final_score": box[4],
                            "size": size,
                            "circum": circum,
                            "size_circum": size / circum.clamp(min=1e-12),
                        }

                    losses = build_faster_rcnn_null_losses_by_stage(
                        rpn_box_coder=model.rpn.box_coder,
                        roi_box_coder=roi_heads.box_coder,
                        rpn_bbox_deltas=rpn_bbox_deltas_img,
                        rpn_anchors=rpn_anchors_img,
                        box_regression=box_regression,
                        pred_img=pred_img,
                        logit_img=logit_img,
                        raw_idx=raw_pred_idx,
                        labels_internal_img=labels_internal_img,
                        proposal_indices_img=proposal_indices_img,
                        proposal_offset=proposal_offset_img,
                        proposals_xyxy=proposals_img,
                        proposal_to_rpn_raw_idx=proposal_to_rpn_raw_idx_img,
                        rpn_objectness_logits=rpn_objectness_img,
                        final_box_xyxy=box[:4],
                        from_size=transformed_images.image_sizes[sample_idx],
                        to_size=original_image_sizes[sample_idx],
                        rpn_bbox_loss=rpn_bbox_loss,
                        rpn_obj_loss=rpn_obj_loss,
                        roi_bbox_loss=roi_bbox_loss,
                        roi_cls_loss=roi_cls_loss,
                        rpn_bbox_direction=rpn_bbox_direction,
                        rpn_obj_direction=rpn_obj_direction,
                        roi_bbox_direction=roi_bbox_direction,
                        roi_cls_direction=roi_cls_direction,
                    )
                    if losses is None:
                        losses = {}
                    rpn_bbox_loss_value = losses.get("rpn_bbox_loss", _zero(device))
                    rpn_obj_loss_value = losses.get("rpn_obj_loss", _zero(device))
                    roi_bbox_loss_value = losses.get("roi_bbox_loss", _zero(device))
                    roi_cls_loss_value = losses.get("roi_cls_loss", _zero(device))
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
                            "xmin": _to_float(box[0]),
                            "ymin": _to_float(box[1]),
                            "xmax": _to_float(box[2]),
                            "ymax": _to_float(box[3]),
                            "score": _to_float(box[4]),
                            "pred_class": pred_class,
                            **{key: _to_float(value) for key, value in prob_values.items()},
                            **{key: _to_float(value) for key, value in shape_values.items()},
                            "rpn_bbox_loss": _to_float(rpn_bbox_loss_value),
                            "rpn_obj_loss": _to_float(rpn_obj_loss_value),
                            "roi_bbox_loss": _to_float(roi_bbox_loss_value),
                            "roi_cls_loss": _to_float(roi_cls_loss_value),
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
            del infer_batch, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
