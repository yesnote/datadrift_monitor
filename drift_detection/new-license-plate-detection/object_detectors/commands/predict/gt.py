from commands.predict.common import *

def run_fn_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    save_image = parsed["save_image_enabled"]
    image_step = parsed["save_image_gt_step"]
    image_max_num = parsed["save_image_gt_max_num"]

    output_csv = run_dir / "fn.csv"
    summary_json = run_dir / "summary.json"

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    total_images = 0
    fn_images = 0
    csv_writer = None
    csv_file_handle = None
    if save_csv:
        csv_file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=["image_id", "image_path", "fn"])
        csv_writer.writeheader()

    try:
        for step_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader))
        ):
            image_list = _as_image_list(images)
            batch_size = len(image_list)
            should_save_step = save_image and (step_idx % image_step == 0)
            step_dir = run_dir / "images" / f"0_{step_idx}"
            if should_save_step:
                step_dir.mkdir(parents=True, exist_ok=True)

            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)

            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                row = {
                    "image_id": int(target["image_id"][0].item()),
                    "image_path": target["path"],
                }

                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                fn = has_fn_for_image(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    iou_match_threshold=iou_match_threshold,
                )
                row["fn"] = fn
                fn_images += int(fn)

                if csv_writer is not None:
                    csv_writer.writerow(row)

                if should_save_step and sample_idx < image_max_num:
                    vis_image = draw_predictions(resized_chws[sample_idx], pred_boxes, pred_class_names, pred_scores)
                    fn_gt_indices = get_fn_gt_indices(
                        gt_boxes=gt_boxes,
                        gt_class_names=gt_class_names,
                        pred_boxes=pred_boxes,
                        pred_class_names=pred_class_names,
                        iou_match_threshold=iou_match_threshold,
                    )
                    for gt_idx in fn_gt_indices:
                        gt_box = gt_boxes[gt_idx]
                        gt_name = gt_class_names[gt_idx]
                        x1, y1, x2, y2 = [int(v) for v in gt_box]
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            vis_image,
                            f"{gt_name}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    out_path = step_dir / f"{row['image_id']}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                total_images += 1
            del infer_batch, preds, _logits, _objectness, _features
    finally:
        if csv_file_handle is not None:
            csv_file_handle.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = _build_summary(total_images, fn_images, output_csv if save_csv else "")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    print(f"Saved summary: {summary_json}")

def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]

    if not save_csv:
        return

    output_csv = run_dir / "tp.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "score",
        "pred_class",
        "max_iou",
        "tp",
    ]

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    raw_prediction,
                    raw_logits,
                    detector.confidence,
                    detector.iou_thresh,
                    classes=None,
                    agnostic=detector.agnostic,
                    return_indices=True,
                )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_boxes_t = det_b[:, :4]
                pred_scores_t = det_b[:, 4] if det_b.shape[1] > 4 else torch.zeros((det_b.shape[0],), device=device)
                pred_cls_ids = det_b[:, 5].long() if det_b.shape[1] > 5 else torch.zeros((det_b.shape[0],), dtype=torch.long, device=device)
                pred_boxes = pred_boxes_t.detach().cpu().tolist()
                pred_scores = pred_scores_t.detach().cpu().tolist()
                pred_class_names = [
                    detector.names[int(c.item())] if detector.names is not None else int(c.item())
                    for c in pred_cls_ids
                ]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)

                tp_flags, best_ious = assign_tp_to_predictions(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=iou_match_threshold,
                )

                for pred_idx, (box, score, pred_class) in enumerate(
                    zip(pred_boxes, pred_scores, pred_class_names)
                ):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                            "score": float(score),
                            "pred_class": pred_class,
                            "max_iou": float(best_ious[pred_idx]),
                            "tp": int(tp_flags[pred_idx]),
                        }
                    )
            del infer_batch, model_output, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")

__all__ = ["run_fn_csv", "run_tp_csv"]
