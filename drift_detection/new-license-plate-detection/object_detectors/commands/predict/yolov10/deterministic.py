from commands.predict.common import *
from commands.predict.yolov10.utils import (
    iter_yolov10_detection_rows,
    parse_yolov10_output_config,
    run_yolov10_forward,
    selected_yolov10_logits,
    selected_yolov10_sigmoid_probs,
)


def run_deterministic_uncertainties_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "deterministic"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    files = {
        "score": (run_dir / "score.csv", ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"]),
        "class_probability": (run_dir / "class_probability.csv", ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"] + [f"prob_{i}" for i in range(num_classes)]),
        "entropy": (run_dir / "entropy.csv", ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "entropy"]),
        "energy": (run_dir / "energy.csv", ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "energy"]),
    }
    handles = {}
    writers = {}
    for key, (path, fieldnames) in files.items():
        handles[key] = open(path, "w", newline="", encoding="utf-8")
        writers[key] = csv.DictWriter(handles[key], fieldnames=fieldnames)
        writers[key].writeheader()
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec"], device=device)
    try:
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch, timing=timing)
            probs_by_sample = {i: selected_yolov10_sigmoid_probs(forward, i, device) for i in range(len(image_list))}
            logits_by_sample = {i: selected_yolov10_logits(forward, i, device) for i in range(len(image_list))}
            batch_items = 0
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                base = dict(item["base_row"])
                writers["score"].writerow(base)
                probs = probs_by_sample[item["sample_idx"]]
                logits = logits_by_sample[item["sample_idx"]]
                row_prob = dict(base)
                values = probs[item["pred_idx"]].detach().cpu().tolist() if item["pred_idx"] < probs.shape[0] else [0.0] * num_classes
                for class_idx in range(num_classes):
                    row_prob[f"prob_{class_idx}"] = float(values[class_idx]) if class_idx < len(values) else 0.0
                writers["class_probability"].writerow(row_prob)
                row_entropy = dict(base)
                row_energy = dict(base)
                if item["pred_idx"] < logits.shape[0]:
                    soft = torch.softmax(logits[item["pred_idx"]], dim=-1)
                    row_entropy["entropy"] = float((-(soft * soft.clamp(min=1e-12).log()).sum()).detach().cpu().item())
                    row_energy["energy"] = float((-torch.logsumexp(logits[item["pred_idx"]], dim=-1)).detach().cpu().item())
                else:
                    row_entropy["entropy"] = 0.0
                    row_energy["energy"] = 0.0
                writers["entropy"].writerow(row_entropy)
                writers["energy"].writerow(row_energy)
                batch_items += 1
            timing.record(len(image_list), batch_items, {"detector_inference_sec": forward.detector_inference_sec})
            del infer_batch, forward
    finally:
        for handle in handles.values():
            handle.close()
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()


__all__ = ["run_deterministic_uncertainties_csv"]
