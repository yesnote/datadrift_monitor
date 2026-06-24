import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, create_dataloader
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.forward import run_faster_rcnn_intermediate_forward
from commands.predict.faster_rcnn.features import tensor_to_float
from commands.predict.faster_rcnn.rows import pred_class_name
from commands.utils.predict_utils import _resize_boxes_xyxy_tensor, build_detector, build_faster_rcnn_null_losses_by_stage


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
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
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

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_faster_rcnn", False)):
        raise NotImplementedError("faster_rcnn null_detect runner requires a Faster R-CNN detector.")

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

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
            result = run_faster_rcnn_intermediate_forward(detector, image_list, device, timing)
            model = detector.detector_model
            roi_heads = model.roi_heads

            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = result.selected_preds[sample_idx] if result.selected_preds and sample_idx < len(result.selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = result.selected_indices[sample_idx] if result.selected_indices and sample_idx < len(result.selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_img = result.raw_prediction[sample_idx]
                logit_img = result.raw_logits[sample_idx] if result.raw_logits is not None else None
                proposals_img = _resize_boxes_xyxy_tensor(
                    result.proposals[sample_idx],
                    result.transformed_images.image_sizes[sample_idx],
                    result.original_image_sizes[sample_idx],
                )
                proposal_to_rpn_raw_idx_img = (
                    result.proposal_to_rpn_raw_indices[sample_idx]
                    if result.proposal_to_rpn_raw_indices and sample_idx < len(result.proposal_to_rpn_raw_indices)
                    else None
                )
                proposal_indices_img = result.proposal_indices_by_img[sample_idx] if sample_idx < len(result.proposal_indices_by_img) else None
                labels_internal_img = result.labels_internal_by_img[sample_idx] if sample_idx < len(result.labels_internal_by_img) else None
                proposal_offset_img = result.proposal_offsets[sample_idx] if sample_idx < len(result.proposal_offsets) else 0
                rpn_objectness_img = result.rpn_objectness_flat[sample_idx]
                rpn_bbox_deltas_img = result.rpn_bbox_deltas_flat[sample_idx]
                rpn_anchors_img = result.rpn_anchors[sample_idx]

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
                        box_regression=result.box_regression,
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
                        from_size=result.transformed_images.image_sizes[sample_idx],
                        to_size=result.original_image_sizes[sample_idx],
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
                    pred_class = pred_class_name(detector, cls_idx)

                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": tensor_to_float(box[0]),
                            "ymin": tensor_to_float(box[1]),
                            "xmax": tensor_to_float(box[2]),
                            "ymax": tensor_to_float(box[3]),
                            "score": tensor_to_float(box[4]),
                            "pred_class": pred_class,
                            **{key: tensor_to_float(value) for key, value in prob_values.items()},
                            **{key: tensor_to_float(value) for key, value in shape_values.items()},
                            "rpn_bbox_loss": tensor_to_float(rpn_bbox_loss_value),
                            "rpn_obj_loss": tensor_to_float(rpn_obj_loss_value),
                            "roi_bbox_loss": tensor_to_float(roi_bbox_loss_value),
                            "roi_cls_loss": tensor_to_float(roi_cls_loss_value),
                        }
                    )

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": result.detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            output_file.flush()
            del result

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
