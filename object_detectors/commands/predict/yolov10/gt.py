from commands.predict.common import *
from commands.predict.yolov10.utils import run_yolov10_forward


def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    if not parsed["save_csv_enabled"] and not parsed["save_image_enabled"]:
        return
    output_csv = run_dir / "tp.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "max_iou", "gt_iou", "tp", "error_type",
    ]
    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    output_file = open(output_csv, "w", newline="", encoding="utf-8") if parsed["save_csv_enabled"] else None
    writer = csv.DictWriter(output_file, fieldnames=fieldnames) if output_file is not None else None
    if writer is not None:
        writer.writeheader()
    try:
        for _step_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader))):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch)
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = forward.selected_preds[sample_idx] if forward.selected_preds and sample_idx < len(forward.selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = forward.selected_indices[sample_idx] if forward.selected_indices and sample_idx < len(forward.selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_boxes = det_b[:, :4].detach().cpu().tolist()
                pred_scores = (det_b[:, 4] if det_b.shape[1] > 4 else torch.zeros((det_b.shape[0],), device=device)).detach().cpu().tolist()
                pred_cls_ids = det_b[:, 5].long() if det_b.shape[1] > 5 else torch.zeros((det_b.shape[0],), dtype=torch.long, device=device)
                pred_class_names = [detector.names[int(c.item())] if detector.names is not None else int(c.item()) for c in pred_cls_ids]
                gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                error_rows = analyze_prediction_error_types(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=parsed["gt_iou_match_threshold"],
                )
                if writer is None:
                    continue
                for pred_idx, row_info in enumerate(error_rows):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else -1
                    box = det_b[pred_idx]
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": float(box[0].detach().cpu().item()),
                            "ymin": float(box[1].detach().cpu().item()),
                            "xmax": float(box[2].detach().cpu().item()),
                            "ymax": float(box[3].detach().cpu().item()),
                            "score": float(box[4].detach().cpu().item()),
                            "pred_class": pred_class_names[pred_idx],
                            "max_iou": row_info["max_iou"],
                            "gt_iou": row_info["gt_iou"],
                            "tp": row_info["tp"],
                            "error_type": row_info["error_type"],
                        }
                    )
            del infer_batch, forward
    finally:
        if output_file is not None:
            output_file.close()
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if parsed["save_csv_enabled"]:
        print(f"Saved results CSV: {output_csv}")


__all__ = ["run_tp_csv"]
