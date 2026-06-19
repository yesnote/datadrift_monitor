from commands.predict.common import *
from commands.predict.yolov10.utils import iter_yolov10_detection_rows, parse_yolov10_output_config, run_yolov10_forward


def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
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
            items_by_sample = {i: [] for i in range(len(image_list))}
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                items_by_sample[item["sample_idx"]].append(item)
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                sample_items = items_by_sample[sample_idx]
                pred_boxes = [item["box"][:4].detach().cpu().tolist() for item in sample_items]
                pred_scores = [float(item["box"][4].detach().cpu().item()) for item in sample_items]
                pred_class_names = [item["base_row"]["pred_class"] for item in sample_items]
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
                for item, row_info in zip(sample_items, error_rows):
                    row = dict(item["base_row"])
                    row.update(
                        {
                            "max_iou": row_info["max_iou"],
                            "gt_iou": row_info["gt_iou"],
                            "tp": row_info["tp"],
                            "error_type": row_info["error_type"],
                        }
                    )
                    writer.writerow(row)
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
