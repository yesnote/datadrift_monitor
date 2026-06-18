from pathlib import Path

from commands.predict.common import *
from commands.predict.yolov10.mc_dropout import _feature_tensor
from commands.predict.yolov10.utils import iter_yolov10_detection_rows, run_yolov10_forward


def _resolve_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    if not parsed["save_csv_enabled"]:
        return
    weights = config.get("output", {}).get("ensemble", {}).get("weights", [])
    if not weights:
        raise ValueError("output.ensemble.weights must contain at least one weight.")
    model_weight = config.get("model", {}).get("weights", "")
    if _resolve_path(weights[0]) != _resolve_path(model_weight):
        raise ValueError("YOLOv10 ensemble requires output.ensemble.weights[0] == model.weights.")
    dataloader = create_dataloader(config, split=split)
    detectors = [build_detector(config, model_weight=w)[0] for w in weights]
    device = next(detectors[0].parameters()).device
    num_classes = len(detectors[0].names) if detectors[0].names is not None else 80
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean", "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std"]
    for class_idx in range(num_classes):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")
    output_csv = run_dir / "ensemble.csv"
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"], device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detectors[0], image_list, device, auto=False)
            detector_inference_sec = 0.0
            feature_compute_sec = 0.0
            with torch.no_grad():
                base = run_yolov10_forward(detectors[0], infer_batch, timing=timing)
            detector_inference_sec += base.detector_inference_sec
            features = []
            for detector in detectors:
                with torch.no_grad():
                    forward = base if detector is detectors[0] else run_yolov10_forward(detector, infer_batch, timing=timing)
                if detector is not detectors[0]:
                    detector_inference_sec += forward.detector_inference_sec
                t_feature = timing.start()
                features.append(_feature_tensor(forward))
                feature_compute_sec += timing.elapsed(t_feature)
            t_feature = timing.start()
            tensor = torch.stack(features, dim=0)
            feat_mean = tensor.mean(dim=0)
            feat_std = tensor.std(dim=0, unbiased=False)
            feature_compute_sec += timing.elapsed(t_feature)
            t_match = timing.start()
            batch_items = 0
            for item in iter_yolov10_detection_rows(detectors[0], targets, base.selected_preds, base.selected_indices, device):
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
            prediction_matching_sec = timing.elapsed(t_match)
            timing.record(len(image_list), batch_items, {"detector_inference_sec": detector_inference_sec, "prediction_matching_sec": prediction_matching_sec, "feature_compute_sec": feature_compute_sec})
            del infer_batch, base, features, tensor, feat_mean, feat_std
    for detector in detectors:
        del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_ensemble_csv"]
