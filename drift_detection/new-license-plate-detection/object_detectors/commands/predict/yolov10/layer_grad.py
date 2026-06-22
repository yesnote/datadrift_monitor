import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import (
    _class_loss_tensor,
    build_detector,
    expand_layer_names,
    map_grad_tensor_to_numbers,
    resolve_layer_parameter,
)


def _bbox_offset_loss(final_xyxy, source_point, mode):
    px, py = source_point[0], source_point[1]
    offset = torch.stack([px - final_xyxy[0], py - final_xyxy[1], final_xyxy[2] - px, final_xyxy[3] - py])
    diff = offset
    if mode == "l1":
        return diff.abs().sum()
    if mode == "l2":
        return diff.pow(2).sum()
    raise ValueError("bbox_loss must be l1 or l2.")


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    layer_cfg = parsed["layer_grad"]
    detector, device = build_detector(config)
    params_by_scalar = {}
    for scalar_name in layer_cfg["scalar"]:
        layers = expand_layer_names(detector.model, layer_cfg["layers_by_scalar"][scalar_name])
        params_by_scalar[scalar_name] = [(layer_name, resolve_layer_parameter(detector.model, layer_name)) for layer_name in layers]
    reductions = layer_cfg["reduction"]
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"]
    for scalar_name in layer_cfg["scalar"]:
        for layer_name, _param in params_by_scalar[scalar_name]:
            safe = layer_name.replace(".", "_")
            for metric in reductions:
                fieldnames.append(f"{safe}_{scalar_name}_{metric}")
    output_csv = run_dir / "layer_grad.csv"
    dataloader = create_dataloader(config, split=split)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=parsed["unit"],
        stages=["detector_inference_sec", "loss_compute_sec", "backpropagation_sec", "feature_compute_sec"],
        device=device,
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            forward = run_yolov10_forward(detector, infer_batch, timing=timing, grad=True)
            loss_compute_sec = 0.0
            backpropagation_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            all_items = list(iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device))
            total_grad_calls = len(all_items) * len(layer_cfg["scalar"])
            grad_call_idx = 0
            for item in all_items:
                row = dict(item["base_row"])
                raw_box_idx = item["raw_box_idx"]
                t_loss = timing.start()
                source_point = forward.source_points[raw_box_idx].to(device=device, dtype=torch.float32)
                final_xyxy = item["box"][:4].float()
                scalar_terms = {}
                if "bbox_loss" in layer_cfg["scalar"]:
                    scalar_terms["bbox_loss"] = _bbox_offset_loss(final_xyxy, source_point, layer_cfg["bbox_loss"])
                if "cls_loss" in layer_cfg["scalar"]:
                    cls_logits = forward.raw_logits[item["sample_idx"], raw_box_idx]
                    target_value = 0.5 if layer_cfg["cls_loss"] == "bcewithlogits" else 1.0 / float(max(1, cls_logits.numel()))
                    cls_target = torch.full_like(cls_logits, target_value)
                    scalar_terms["cls_loss"] = _class_loss_tensor(
                        cls_logits,
                        cls_target,
                        class_idx=None,
                        mode=layer_cfg["cls_loss"],
                        direction=layer_cfg["cls_direction"],
                        reduction="sum",
                    )
                loss_compute_sec += timing.elapsed(t_loss)
                for scalar_name, target_scalar in scalar_terms.items():
                    grad_call_idx += 1
                    t_back = timing.start()
                    grads = torch.autograd.grad(
                        target_scalar,
                        [param for _layer_name, param in params_by_scalar[scalar_name]],
                        retain_graph=grad_call_idx < total_grad_calls,
                    )
                    backpropagation_sec += timing.elapsed(t_back)
                    t_feature = timing.start()
                    for (layer_name, _param), grad in zip(params_by_scalar[scalar_name], grads):
                        safe = layer_name.replace(".", "_")
                        stats = map_grad_tensor_to_numbers(grad)
                        for metric in reductions:
                            value = stats.get(metric, 0.0)
                            row[f"{safe}_{scalar_name}_{metric}"] = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                    feature_compute_sec += timing.elapsed(t_feature)
                writer.writerow(row)
                batch_items += 1
            timing.record(
                len(image_list),
                batch_items,
                {
                    "detector_inference_sec": forward.detector_inference_sec,
                    "loss_compute_sec": loss_compute_sec,
                    "backpropagation_sec": backpropagation_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del infer_batch, forward, all_items
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_layer_grad_csv"]
