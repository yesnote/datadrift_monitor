from commands.predict.common import *
from commands.utils.predict_utils import _class_loss_tensor, expand_layer_names, map_grad_tensor_to_numbers, resolve_layer_parameter
from commands.predict.yolov10.utils import (
    iter_yolov10_detection_rows,
    parse_yolov10_output_config,
    run_yolov10_forward,
)


def _parse_config(config):
    output = config.get("output", {})
    layer_cfg = output.get("layer_grad", {}) if isinstance(output.get("layer_grad", {}), dict) else {}
    grad = layer_cfg.get("gradient", {}) if isinstance(layer_cfg.get("gradient", {}), dict) else {}
    target = str(grad.get("target", "null_target")).strip().lower()
    if target != "null_target":
        raise NotImplementedError("YOLOv10 layer_grad supports only target=null_target.")
    scalar = grad.get("scalar", ["bbox_loss", "cls_loss"])
    scalar = [str(v).strip().lower() for v in (scalar if isinstance(scalar, (list, tuple)) else [scalar])]
    if "loss" in scalar:
        scalar = ["bbox_loss", "cls_loss"]
    for value in scalar:
        if value not in {"bbox_loss", "cls_loss"}:
            raise ValueError("YOLOv10 layer_grad.gradient.scalar supports bbox_loss and cls_loss only.")
    bbox_loss = str(grad.get("bbox_loss", "l1")).strip().lower()
    if bbox_loss not in {"l1", "l2"}:
        raise ValueError("YOLOv10 layer_grad.gradient.bbox_loss supports only l1 or l2.")
    cls_loss = str(grad.get("cls_loss", "bcewithlogits")).strip().lower()
    if cls_loss not in {"bcewithlogits", "kl"}:
        raise ValueError("YOLOv10 layer_grad.gradient.cls_loss supports only bcewithlogits or kl.")
    reduction = grad.get("reduction", ["l1_norm", "l2_norm", "min", "max", "mean", "std"])
    reduction = [str(v).strip() for v in (reduction if isinstance(reduction, (list, tuple)) else [reduction]) if str(v).strip()]
    if not reduction:
        raise ValueError("YOLOv10 layer_grad requires reduction metrics; raw gradient saving is not supported.")
    layers = grad.get("layer", ["model.23.one2one_cv2.0.2", "model.23.one2one_cv3.0.2"])
    layer_list = [str(v).strip() for v in (layers if isinstance(layers, (list, tuple)) else [layers]) if str(v).strip()]
    for layer_name in layer_list:
        if not (layer_name.startswith("model.23.one2one_cv2.") or layer_name.startswith("model.23.one2one_cv3.")):
            raise ValueError("YOLOv10 layer_grad supports only one-to-one head layers: model.23.one2one_cv2.* or model.23.one2one_cv3.*")
    return {
        "scalar": scalar,
        "layer": layer_list,
        "reduction": reduction,
        "bbox_loss": bbox_loss,
        "cls_loss": cls_loss,
        "bbox_direction": str(grad.get("bbox_direction", "pred_to_target")).strip().lower(),
        "cls_direction": str(grad.get("cls_direction", "pred_to_target")).strip().lower(),
    }


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
    layer_cfg = _parse_config(config)
    detector, device = build_detector(config)
    layers = expand_layer_names(detector.model, layer_cfg["layer"])
    params = [(layer_name, resolve_layer_parameter(detector.model, layer_name)) for layer_name in layers]
    reductions = layer_cfg["reduction"]
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"]
    for layer_name, _param in params:
        safe = layer_name.replace(".", "_")
        for scalar_name in layer_cfg["scalar"]:
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
            detector.zero_grad(set_to_none=True)
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
                        [param for _layer_name, param in params],
                        retain_graph=grad_call_idx < total_grad_calls,
                        allow_unused=True,
                    )
                    if not any(grad is not None for grad in grads):
                        raise RuntimeError(
                            "YOLOv10 layer_grad produced no gradient for the selected scalar. "
                            f"scalar={scalar_name}, layers={layer_cfg['layer']}"
                        )
                    backpropagation_sec += timing.elapsed(t_back)
                    t_feature = timing.start()
                    for (layer_name, _param), grad in zip(params, grads):
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_layer_grad_csv"]
