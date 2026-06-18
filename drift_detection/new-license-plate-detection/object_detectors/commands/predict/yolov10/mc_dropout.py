from commands.predict.common import *
from commands.predict.yolov10.utils import iter_yolov10_detection_rows, run_yolov10_forward
from models.yolov10.core import xywh2xyxy


def _feature_tensor(forward):
    boxes = xywh2xyxy(forward.decoded_prediction[..., :4].detach().float())
    probs = torch.sigmoid(forward.raw_logits.detach().float())
    scores = probs.max(dim=-1, keepdim=True).values
    return torch.cat([boxes, scores, probs], dim=-1)


def _set_dropout(model, rate):
    handles = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()
            module.p = float(rate)
            handles.append(module)
    return handles


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    if not parsed["save_csv_enabled"]:
        return
    num_runs = int(parsed["mc_num_runs"])
    if num_runs <= 0:
        raise ValueError("mc_dropout.num_runs must be positive.")
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean", "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std"]
    for class_idx in range(num_classes):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")
    output_csv = run_dir / "mc_dropout.csv"
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"], device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            detector_inference_sec = 0.0
            feature_compute_sec = 0.0
            prediction_matching_sec = 0.0
            with torch.no_grad():
                base = run_yolov10_forward(detector, infer_batch, timing=timing)
            detector_inference_sec += base.detector_inference_sec
            runs = []
            dropout_modules = _set_dropout(detector.model, parsed["mc_dropout_rate"])
            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        run = run_yolov10_forward(detector, infer_batch, timing=timing)
                        detector_inference_sec += run.detector_inference_sec
                        t_feature = timing.start()
                        runs.append(_feature_tensor(run))
                        feature_compute_sec += timing.elapsed(t_feature)
            finally:
                detector.model.eval()
                del dropout_modules
            t_feature = timing.start()
            runs_tensor = torch.stack(runs, dim=0)
            feat_mean = runs_tensor.mean(dim=0)
            feat_std = runs_tensor.std(dim=0, unbiased=False)
            feature_compute_sec += timing.elapsed(t_feature)
            t_match = timing.start()
            batch_items = 0
            for item in iter_yolov10_detection_rows(detector, targets, base.selected_preds, base.selected_indices, device):
                raw_idx = item["raw_pred_idx"]
                mean_vec = feat_mean[item["sample_idx"], raw_idx].detach().cpu()
                std_vec = feat_std[item["sample_idx"], raw_idx].detach().cpu()
                row = dict(item["base_row"])
                row.update(
                    {
                        "xmin_mean": float(mean_vec[0].item()),
                        "ymin_mean": float(mean_vec[1].item()),
                        "xmax_mean": float(mean_vec[2].item()),
                        "ymax_mean": float(mean_vec[3].item()),
                        "score_mean": float(mean_vec[4].item()),
                        "xmin_std": float(std_vec[0].item()),
                        "ymin_std": float(std_vec[1].item()),
                        "xmax_std": float(std_vec[2].item()),
                        "ymax_std": float(std_vec[3].item()),
                        "score_std": float(std_vec[4].item()),
                    }
                )
                for class_idx in range(num_classes):
                    row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item()) if class_idx < mean_vec.numel() - 5 else 0.0
                    row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item()) if class_idx < std_vec.numel() - 5 else 0.0
                writer.writerow(row)
                batch_items += 1
            prediction_matching_sec += timing.elapsed(t_match)
            timing.record(len(image_list), batch_items, {"detector_inference_sec": detector_inference_sec, "prediction_matching_sec": prediction_matching_sec, "feature_compute_sec": feature_compute_sec})
            del infer_batch, base, runs, runs_tensor, feat_mean, feat_std
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_mc_dropout_csv"]
