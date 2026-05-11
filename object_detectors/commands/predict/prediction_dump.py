from commands.predict.common import *


PREDICTION_DUMP_GRAD_TARGETS = [
    "score_null_diff",
    "obj_null_bce_loss",
    "cls_uniform_kl",
    "bbox_anchor_log_area_ratio",
]


def _safe_layer_col_name(layer_name):
    return str(layer_name).replace(".", "_").replace("/", "_").replace("\\", "_")


def _grad_norm_col(target_name, layer_name):
    return f"{target_name}_grad_norm_{_safe_layer_col_name(layer_name)}"


class _LayerOutputGradNormBuffer:
    def __init__(self, model, target_layers):
        self.target_layers = list(target_layers)
        modules = dict(model.named_modules())
        self.handles = []
        self.norms = {}
        for layer_name in self.target_layers:
            if layer_name not in modules:
                raise ValueError(f"Gradient layer not found in model.named_modules(): {layer_name}")
            module = modules[layer_name]
            if hasattr(module, "register_full_backward_hook"):
                self.handles.append(
                    module.register_full_backward_hook(
                        lambda _module, _grad_input, grad_output, name=layer_name: self._hook(name, grad_output)
                    )
                )
            else:
                self.handles.append(
                    module.register_backward_hook(
                        lambda _module, _grad_input, grad_output, name=layer_name: self._hook(name, grad_output)
                    )
                )

    def _hook(self, layer_name, grad_output):
        if grad_output is None or len(grad_output) == 0 or grad_output[0] is None:
            return None
        grad = grad_output[0].detach()
        norm = float(torch.linalg.vector_norm(grad.float()).detach().cpu().item())
        self.norms[layer_name] = norm
        return None

    def clear(self):
        self.norms.clear()

    def values(self):
        return {layer_name: float(self.norms.get(layer_name, 0.0)) for layer_name in self.target_layers}

    def remove(self):
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _class_name(names, cls_idx):
    if isinstance(names, dict):
        return names.get(cls_idx, str(cls_idx))
    if isinstance(names, list) and 0 <= cls_idx < len(names):
        return names[cls_idx]
    return str(cls_idx)


def _prediction_dump_gradient_config(config):
    pd_cfg = ((config.get("output", {}) or {}).get("prediction_dump", {}) or {})
    grad_cfg = pd_cfg.get("gradient", {}) or {}
    enabled = bool(grad_cfg.get("enabled", False))
    layers = grad_cfg.get("layer", ["model.24.m.0", "model.24.m.1", "model.24.m.2"])
    if isinstance(layers, str):
        layers = [layers]
    layers = [str(v).strip() for v in layers if str(v).strip()]
    return enabled, layers


def _prob_distribution_for_grad(raw_row, logit_row=None):
    if logit_row is not None and getattr(logit_row, "numel", lambda: 0)() > 0:
        return torch.softmax(logit_row.float(), dim=-1)
    if raw_row is not None and raw_row.shape[0] > 5:
        probs = raw_row[5:].float().clamp(min=1e-12)
        return probs / probs.sum().clamp(min=1e-12)
    device = raw_row.device if raw_row is not None else "cpu"
    return torch.zeros((0,), dtype=torch.float32, device=device)


def _null_gradient_scalars(raw_row, logit_row, cls_idx, prior_row):
    eps = 1e-12
    out = {}
    device = raw_row.device
    obj = raw_row[4].float().clamp(min=1e-6, max=1.0 - 1e-6) if raw_row.shape[0] > 4 else torch.ones((), device=device)
    probs = _prob_distribution_for_grad(raw_row, logit_row)
    n_cls = int(probs.shape[0]) if probs.numel() else 0
    cls_conf = probs[int(cls_idx)].clamp(min=eps) if cls_idx >= 0 and n_cls > int(cls_idx) else torch.zeros((), device=device)
    null_score = 0.5 * ((1.0 / float(n_cls)) if n_cls > 0 else 0.0)
    out["score_null_diff"] = (obj * cls_conf) - float(null_score)
    out["obj_null_bce_loss"] = -(0.5 * torch.log(obj) + 0.5 * torch.log(1.0 - obj))
    if n_cls > 0:
        entropy = -(probs.clamp(min=eps) * torch.log(probs.clamp(min=eps))).sum()
        out["cls_uniform_kl"] = torch.log(torch.tensor(float(n_cls), dtype=probs.dtype, device=probs.device)) - entropy
    else:
        out["cls_uniform_kl"] = raw_row.sum() * 0.0
    if prior_row is not None and prior_row.shape[0] >= 4 and raw_row.shape[0] >= 4:
        pred_w = raw_row[2].float().clamp(min=eps)
        pred_h = raw_row[3].float().clamp(min=eps)
        anchor_w = prior_row[2].float().clamp(min=eps)
        anchor_h = prior_row[3].float().clamp(min=eps)
        out["bbox_anchor_log_area_ratio"] = torch.log((pred_w * pred_h) / (anchor_w * anchor_h).clamp(min=eps))
    else:
        out["bbox_anchor_log_area_ratio"] = raw_row[:4].sum() * 0.0
    return out


def _collect_null_gradient_norms(detector, grad_buffer, target_layers, raw_row, logit_row, cls_idx, prior_row):
    out = {}
    for target_name in PREDICTION_DUMP_GRAD_TARGETS:
        for layer_name in target_layers:
            out[_grad_norm_col(target_name, layer_name)] = 0.0
    scalars = _null_gradient_scalars(raw_row, logit_row, cls_idx, prior_row)
    for target_name in PREDICTION_DUMP_GRAD_TARGETS:
        scalar = scalars.get(target_name)
        if scalar is None or not getattr(scalar, "requires_grad", False):
            continue
        detector.zero_grad(set_to_none=True)
        grad_buffer.clear()
        scalar.backward(retain_graph=True)
        for layer_name, norm in grad_buffer.values().items():
            out[_grad_norm_col(target_name, layer_name)] = norm
    detector.zero_grad(set_to_none=True)
    grad_buffer.clear()
    return out


def _null_target_stats(raw_row, logit_row, cls_idx, obj, score):
    if logit_row is not None and getattr(logit_row, "numel", lambda: 0)() > 0:
        cls_probs = torch.softmax(logit_row.detach().float(), dim=-1)
    elif raw_row is not None and raw_row.shape[0] > 5:
        cls_probs = raw_row[5:].detach().float().clamp(min=0.0)
        prob_sum = cls_probs.sum()
        if float(prob_sum.detach().cpu().item()) > 1e-12:
            cls_probs = cls_probs / prob_sum
    else:
        cls_probs = torch.zeros((0,), dtype=torch.float32)

    n_cls = int(cls_probs.shape[0]) if cls_probs.numel() else 0
    null_obj = 0.5
    null_cls_conf = (1.0 / float(n_cls)) if n_cls > 0 else 0.0
    null_score = null_obj * null_cls_conf
    obj_prob = min(max(float(obj), 1e-6), 1.0 - 1e-6)
    obj_null_bce_loss = -(
        null_obj * np.log(obj_prob) + (1.0 - null_obj) * np.log(1.0 - obj_prob)
    )
    cls_prob = (
        float(cls_probs[cls_idx].detach().cpu().item())
        if cls_idx >= 0 and n_cls > cls_idx
        else 0.0
    )
    if n_cls > 0:
        eps = 1e-12
        probs = cls_probs.clamp(min=eps)
        entropy = float((-(probs * torch.log(probs)).sum()).detach().cpu().item())
        max_entropy = float(np.log(float(n_cls)))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        uniform_kl = max_entropy - entropy
    else:
        entropy = 0.0
        entropy_norm = 0.0
        uniform_kl = 0.0
    return {
        "null_obj": null_obj,
        "null_cls_conf": null_cls_conf,
        "null_score": null_score,
        "obj_null_abs_diff": abs(float(obj) - null_obj),
        "obj_null_bce_loss": float(obj_null_bce_loss),
        "cls_null_abs_diff": abs(float(cls_prob) - null_cls_conf),
        "score_null_diff": float(score) - null_score,
        "cls_entropy": entropy,
        "cls_entropy_norm": entropy_norm,
        "cls_uniform_kl": uniform_kl,
    }


def _ciou_xyxy(box_a, box_b, eps=1e-7):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    area_a = aw * ah
    area_b = bw * bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = area_a + area_b - inter + eps
    iou = inter / union

    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    cw = max(ax2, bx2) - min(ax1, bx1)
    ch = max(ay2, by2) - min(ay1, by1)
    c2 = cw ** 2 + ch ** 2 + eps
    v = (4.0 / (np.pi ** 2)) * (np.arctan(bw / max(bh, eps)) - np.arctan(aw / max(ah, eps))) ** 2
    alpha = v / max(1.0 - iou + v, eps)
    return float(iou - (rho2 / c2 + alpha * v))


def _prob_distribution(raw_row, logit_row=None):
    if logit_row is not None and getattr(logit_row, "numel", lambda: 0)() > 0:
        return torch.softmax(logit_row.detach().float(), dim=-1)
    if raw_row is not None and raw_row.shape[0] > 5:
        probs = raw_row[5:].detach().float().clamp(min=0.0)
        prob_sum = probs.sum()
        if float(prob_sum.detach().cpu().item()) > 1e-12:
            return probs / prob_sum
        return torch.zeros_like(probs)
    return torch.zeros((0,), dtype=torch.float32)


def _iou_1vN_xyxy(box: torch.Tensor, boxes: torch.Tensor):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=box.device)
    lt = torch.max(box[:2], boxes[:, :2])
    rb = torch.min(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-12)


def _cand_target_stats(
    raw_b,
    logit_b,
    raw_pred_idx,
    cls_idx,
    pred_box_xyxy,
    pred_obj,
    pred_score,
    pred_area,
    iou_threshold,
    score_threshold=0.1,
):
    eps = 1e-12
    zero = {
        "score_cand_diff": 0.0,
        "obj_cand_bce_loss": 0.0,
        "cls_cand_kl": 0.0,
        "bbox_cand_log_area_ratio": 0.0,
        "bbox_cand_log_area_ratio_std": 0.0,
        "num_cand_boxes": 0,
        "num_nonself_cand_boxes": 0,
        "has_nonself_cand": 0,
        "cand_score_mean": 0.0,
        "cand_score_min": 0.0,
        "cand_score_max": 0.0,
        "cand_score_std": 0.0,
        "cand_iou_mean": 0.0,
        "cand_iou_min": 0.0,
        "cand_iou_max": 0.0,
        "cand_iou_std": 0.0,
        "cand_area_mean": 0.0,
        "cand_area_min": 0.0,
        "cand_area_max": 0.0,
        "cand_area_std": 0.0,
        "cand_score_threshold": float(score_threshold),
        "cand_iou_threshold": float(iou_threshold),
    }
    if raw_b is None or raw_b.numel() == 0 or raw_pred_idx < 0 or raw_pred_idx >= int(raw_b.shape[0]):
        return zero

    raw_xyxy = _xywh_to_xyxy_tensor(raw_b[:, :4])
    ious = _iou_1vN_xyxy(pred_box_xyxy, raw_xyxy)
    raw_obj = raw_b[:, 4] if raw_b.shape[1] > 4 else torch.ones((raw_b.shape[0],), device=raw_b.device)
    raw_cls = raw_b[:, 5:] if raw_b.shape[1] > 5 else torch.zeros((raw_b.shape[0], 0), device=raw_b.device)
    if raw_cls.numel() > 0:
        raw_cls_max, raw_cls_idx = raw_cls.max(dim=1)
    else:
        raw_cls_max = torch.ones_like(raw_obj)
        raw_cls_idx = torch.zeros((raw_b.shape[0],), dtype=torch.long, device=raw_b.device)
    raw_score = raw_obj * raw_cls_max
    class_mask = (raw_cls_idx == int(cls_idx)) if cls_idx >= 0 else torch.ones_like(raw_score, dtype=torch.bool)
    cand_mask = class_mask & (ious >= float(iou_threshold)) & (raw_score >= float(score_threshold))
    cand_mask[raw_pred_idx] = False

    has_nonself_cand = bool(cand_mask.any())
    if has_nonself_cand:
        cand_indices = cand_mask.nonzero(as_tuple=False).view(-1)
    else:
        cand_indices = torch.tensor([int(raw_pred_idx)], dtype=torch.long, device=raw_b.device)

    pred_dist = _prob_distribution(raw_b[raw_pred_idx], logit_b[raw_pred_idx] if logit_b is not None and raw_pred_idx < int(logit_b.shape[0]) else None)
    pred_obj_clamped = min(max(float(pred_obj), 1e-6), 1.0 - 1e-6)
    score_diffs = []
    obj_bces = []
    cls_kls = []
    area_ratios = []
    cand_scores = []
    cand_ious = []
    cand_areas = []
    for cand_idx_tensor in cand_indices:
        cand_idx = int(cand_idx_tensor.detach().cpu().item())
        cand_row = raw_b[cand_idx]
        cand_obj = float(cand_row[4].detach().cpu().item()) if cand_row.shape[0] > 4 else 1.0
        cand_obj_clamped = min(max(cand_obj, 1e-6), 1.0 - 1e-6)
        obj_bces.append(
            -(
                cand_obj_clamped * np.log(pred_obj_clamped)
                + (1.0 - cand_obj_clamped) * np.log(1.0 - pred_obj_clamped)
            )
        )

        cand_dist = _prob_distribution(cand_row, logit_b[cand_idx] if logit_b is not None and cand_idx < int(logit_b.shape[0]) else None)
        if pred_dist.numel() and cand_dist.numel() and int(pred_dist.shape[0]) == int(cand_dist.shape[0]):
            pred_probs = pred_dist.clamp(min=eps)
            cand_probs = cand_dist.clamp(min=eps)
            cls_kls.append(float((cand_probs * (torch.log(cand_probs) - torch.log(pred_probs))).sum().detach().cpu().item()))
        else:
            cls_kls.append(0.0)

        cand_xyxy = raw_xyxy[cand_idx]
        cand_w = max(0.0, float((cand_xyxy[2] - cand_xyxy[0]).detach().cpu().item()))
        cand_h = max(0.0, float((cand_xyxy[3] - cand_xyxy[1]).detach().cpu().item()))
        cand_area = cand_w * cand_h
        cand_score = float(raw_score[cand_idx].detach().cpu().item())
        cand_iou = float(ious[cand_idx].detach().cpu().item())
        score_diffs.append(float(pred_score) - cand_score)
        area_ratios.append(float(np.log(max(float(pred_area), eps) / max(float(cand_area), eps))))
        cand_scores.append(cand_score)
        cand_ious.append(cand_iou)
        cand_areas.append(cand_area)

    cand_scores_arr = np.asarray(cand_scores, dtype=np.float64)
    cand_ious_arr = np.asarray(cand_ious, dtype=np.float64)
    cand_areas_arr = np.asarray(cand_areas, dtype=np.float64)
    area_ratios_arr = np.asarray(area_ratios, dtype=np.float64)

    return {
        "score_cand_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "obj_cand_bce_loss": float(np.mean(obj_bces)) if obj_bces else 0.0,
        "cls_cand_kl": float(np.mean(cls_kls)) if cls_kls else 0.0,
        "bbox_cand_log_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "bbox_cand_log_area_ratio_std": float(np.std(area_ratios_arr)) if area_ratios_arr.size else 0.0,
        "num_cand_boxes": int(cand_indices.numel()),
        "num_nonself_cand_boxes": int(cand_indices.numel()) if has_nonself_cand else 0,
        "has_nonself_cand": int(has_nonself_cand),
        "cand_score_mean": float(np.mean(cand_scores_arr)) if cand_scores_arr.size else 0.0,
        "cand_score_min": float(np.min(cand_scores_arr)) if cand_scores_arr.size else 0.0,
        "cand_score_max": float(np.max(cand_scores_arr)) if cand_scores_arr.size else 0.0,
        "cand_score_std": float(np.std(cand_scores_arr)) if cand_scores_arr.size else 0.0,
        "cand_iou_mean": float(np.mean(cand_ious_arr)) if cand_ious_arr.size else 0.0,
        "cand_iou_min": float(np.min(cand_ious_arr)) if cand_ious_arr.size else 0.0,
        "cand_iou_max": float(np.max(cand_ious_arr)) if cand_ious_arr.size else 0.0,
        "cand_iou_std": float(np.std(cand_ious_arr)) if cand_ious_arr.size else 0.0,
        "cand_area_mean": float(np.mean(cand_areas_arr)) if cand_areas_arr.size else 0.0,
        "cand_area_min": float(np.min(cand_areas_arr)) if cand_areas_arr.size else 0.0,
        "cand_area_max": float(np.max(cand_areas_arr)) if cand_areas_arr.size else 0.0,
        "cand_area_std": float(np.std(cand_areas_arr)) if cand_areas_arr.size else 0.0,
        "cand_score_threshold": float(score_threshold),
        "cand_iou_threshold": float(iou_threshold),
    }


def run_prediction_dump_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "prediction_dump"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    if unit != "bbox":
        raise ValueError("output.uncertainty='prediction_dump' requires output.unit='bbox'.")
    if not save_csv:
        return

    output_csv = run_dir / "prediction_dump.csv"
    gradient_enabled, gradient_layers = _prediction_dump_gradient_config(config)
    gradient_columns = [
        _grad_norm_col(target_name, layer_name)
        for layer_name in gradient_layers
        for target_name in PREDICTION_DUMP_GRAD_TARGETS
    ]
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "cx",
        "cy",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "anchor_cx",
        "anchor_cy",
        "anchor_w",
        "anchor_h",
        "anchor_area",
        "anchor_aspect_ratio",
        "bbox_anchor_dx",
        "bbox_anchor_dy",
        "bbox_anchor_center_l2",
        "bbox_anchor_w_ratio",
        "bbox_anchor_h_ratio",
        "bbox_anchor_area_ratio",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_aspect_ratio_diff",
        "bbox_anchor_ciou",
        "bbox_anchor_ciou_loss",
        "obj",
        "cls_conf",
        "score",
        "pred_class",
        "pred_class_id",
        "null_obj",
        "null_cls_conf",
        "null_score",
        "obj_null_abs_diff",
        "obj_null_bce_loss",
        "cls_null_abs_diff",
        "score_null_diff",
        "cls_entropy",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "score_cand_diff",
        "obj_cand_bce_loss",
        "cls_cand_kl",
        "bbox_cand_log_area_ratio",
        "bbox_cand_log_area_ratio_std",
        "num_cand_boxes",
        "num_nonself_cand_boxes",
        "has_nonself_cand",
        "cand_score_mean",
        "cand_score_min",
        "cand_score_max",
        "cand_score_std",
        "cand_iou_mean",
        "cand_iou_min",
        "cand_iou_max",
        "cand_iou_std",
        "cand_area_mean",
        "cand_area_min",
        "cand_area_max",
        "cand_area_std",
        "cand_score_threshold",
        "cand_iou_threshold",
        *gradient_columns,
        "max_iou",
        "tp",
    ]

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    grad_buffer = _LayerOutputGradNormBuffer(detector.model, gradient_layers) if gradient_enabled else None
    raw_prof = RawComputeProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=unit)
    total_images = 0
    total_predictions = 0

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for images, targets in tqdm(
                dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
            ):
                image_list = _as_image_list(images)
                total_images += len(image_list)
                detector.zero_grad(set_to_none=True)
                infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
                if gradient_enabled:
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                    raw_logits = (
                        model_output[1]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                        else None
                    )
                    raw_priors = (
                        model_output[3]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 3
                        else None
                    )
                    nms_prediction = raw_prediction.detach()
                    nms_raw_logits = raw_logits.detach() if raw_logits is not None else None
                    nms_logits = _resolve_nms_logits(nms_prediction, nms_raw_logits)
                    selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                        prediction=nms_prediction,
                        logits=nms_logits,
                        conf_thres=nms_kwargs["conf_thres"],
                        iou_thres=nms_kwargs["iou_thres"],
                        classes=nms_kwargs["classes"],
                        agnostic=nms_kwargs["agnostic"],
                        max_det=nms_kwargs["max_det"],
                        return_indices=True,
                    )
                else:
                    with torch.no_grad():
                        model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = (
                            model_output[1]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                            else None
                        )
                        raw_priors = (
                            model_output[3]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 3
                            else None
                        )
                        nms_logits = _resolve_nms_logits(raw_prediction, raw_logits)
                        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                            prediction=raw_prediction,
                            logits=nms_logits,
                            conf_thres=nms_kwargs["conf_thres"],
                            iou_thres=nms_kwargs["iou_thres"],
                            classes=nms_kwargs["classes"],
                            agnostic=nms_kwargs["agnostic"],
                            max_det=nms_kwargs["max_det"],
                            return_indices=True,
                        )

                t_raw = raw_prof.start()
                batch_items = 0
                for sample_idx in range(len(image_list)):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    det = selected_preds[sample_idx]
                    raw_keep_b = selected_indices[sample_idx]
                    raw_b_grad = raw_prediction[sample_idx].float()
                    raw_b = raw_b_grad.detach()
                    pred_boxes = det[:, :4].detach().cpu().tolist()
                    pred_scores = det[:, 4].detach().cpu().tolist() if det.shape[1] > 4 else []
                    pred_cls_ids = (
                        det[:, 5].long().detach().cpu().tolist()
                        if det.shape[1] > 5
                        else [0 for _ in range(int(det.shape[0]))]
                    )
                    pred_class_names = [_class_name(detector.names, int(cls_idx)) for cls_idx in pred_cls_ids]
                    gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                    gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                    tp_flags, best_ious = assign_tp_to_predictions(
                        gt_boxes=gt_boxes,
                        gt_class_names=gt_class_names,
                        pred_boxes=pred_boxes,
                        pred_class_names=pred_class_names,
                        pred_scores=pred_scores,
                        iou_match_threshold=iou_match_threshold,
                    )
                    batch_items += int(det.shape[0])
                    total_predictions += int(det.shape[0])

                    for pred_idx, box in enumerate(det):
                        raw_pred_idx = (
                            int(raw_keep_b[pred_idx].detach().cpu().item())
                            if pred_idx < int(raw_keep_b.shape[0])
                            else pred_idx
                        )
                        cls_idx = int(pred_cls_ids[pred_idx]) if pred_idx < len(pred_cls_ids) else -1
                        xmin = float(box[0].detach().cpu().item())
                        ymin = float(box[1].detach().cpu().item())
                        xmax = float(box[2].detach().cpu().item())
                        ymax = float(box[3].detach().cpu().item())
                        w = max(0.0, xmax - xmin)
                        h = max(0.0, ymax - ymin)
                        area = w * h
                        obj = 0.0
                        cls_conf = 0.0
                        raw_row = None
                        raw_row_grad = None
                        logit_row = None
                        logit_row_grad = None
                        prior_row = None
                        prior_row_grad = None
                        if 0 <= raw_pred_idx < int(raw_b.shape[0]):
                            raw_row = raw_b[raw_pred_idx]
                            raw_row_grad = raw_b_grad[raw_pred_idx]
                            obj = float(raw_row[4].detach().cpu().item()) if raw_row.shape[0] > 4 else 1.0
                            if cls_idx >= 0 and raw_row.shape[0] > 5 + cls_idx:
                                cls_conf = float(raw_row[5 + cls_idx].detach().cpu().item())
                        if raw_logits is not None and sample_idx < int(raw_logits.shape[0]) and 0 <= raw_pred_idx < int(raw_logits.shape[1]):
                            logit_row_grad = raw_logits[sample_idx, raw_pred_idx]
                            logit_row = logit_row_grad.detach()
                        if raw_priors is not None and sample_idx < int(raw_priors.shape[0]) and 0 <= raw_pred_idx < int(raw_priors.shape[1]):
                            prior_row_grad = raw_priors[sample_idx, raw_pred_idx].float()
                            prior_row = prior_row_grad.detach()
                        score = float(box[4].detach().cpu().item())
                        if cls_conf == 0.0 and obj > 1e-12:
                            cls_conf = score / obj
                        null_stats = _null_target_stats(raw_row, logit_row, cls_idx, obj, score)
                        logit_b = raw_logits[sample_idx] if raw_logits is not None and sample_idx < int(raw_logits.shape[0]) else None
                        cand_stats = _cand_target_stats(
                            raw_b=raw_b,
                            logit_b=logit_b.detach() if logit_b is not None else None,
                            raw_pred_idx=raw_pred_idx,
                            cls_idx=cls_idx,
                            pred_box_xyxy=box[:4].detach().float(),
                            pred_obj=obj,
                            pred_score=score,
                            pred_area=area,
                            iou_threshold=nms_kwargs["iou_thres"],
                            score_threshold=0.1,
                        )
                        gradient_stats = {col: 0.0 for col in gradient_columns}
                        if gradient_enabled and grad_buffer is not None and raw_row_grad is not None:
                            gradient_stats = _collect_null_gradient_norms(
                                detector=detector,
                                grad_buffer=grad_buffer,
                                target_layers=gradient_layers,
                                raw_row=raw_row_grad,
                                logit_row=logit_row_grad,
                                cls_idx=cls_idx,
                                prior_row=prior_row_grad,
                            )
                        anchor_cx = anchor_cy = anchor_w = anchor_h = 0.0
                        if prior_row is not None and prior_row.shape[0] >= 4:
                            anchor_cx = float(prior_row[0].detach().cpu().item())
                            anchor_cy = float(prior_row[1].detach().cpu().item())
                            anchor_w = float(prior_row[2].detach().cpu().item())
                            anchor_h = float(prior_row[3].detach().cpu().item())
                        anchor_area = max(0.0, anchor_w) * max(0.0, anchor_h)
                        anchor_aspect_ratio = anchor_w / max(anchor_h, 1e-12)
                        cx = 0.5 * (xmin + xmax)
                        cy = 0.5 * (ymin + ymax)
                        dx = cx - anchor_cx
                        dy = cy - anchor_cy
                        w_ratio = w / max(anchor_w, 1e-12)
                        h_ratio = h / max(anchor_h, 1e-12)
                        area_ratio = area / max(anchor_area, 1e-12)
                        anchor_xmin = anchor_cx - 0.5 * anchor_w
                        anchor_ymin = anchor_cy - 0.5 * anchor_h
                        anchor_xmax = anchor_cx + 0.5 * anchor_w
                        anchor_ymax = anchor_cy + 0.5 * anchor_h
                        bbox_anchor_ciou = _ciou_xyxy(
                            (xmin, ymin, xmax, ymax),
                            (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax),
                        )

                        writer.writerow(
                            {
                                "image_id": image_id,
                                "image_path": image_path,
                                "pred_idx": pred_idx,
                                "raw_pred_idx": raw_pred_idx,
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax,
                                "cx": cx,
                                "cy": cy,
                                "w": w,
                                "h": h,
                                "area": area,
                                "aspect_ratio": w / max(h, 1e-12),
                                "anchor_cx": anchor_cx,
                                "anchor_cy": anchor_cy,
                                "anchor_w": anchor_w,
                                "anchor_h": anchor_h,
                                "anchor_area": anchor_area,
                                "anchor_aspect_ratio": anchor_aspect_ratio,
                                "bbox_anchor_dx": dx,
                                "bbox_anchor_dy": dy,
                                "bbox_anchor_center_l2": float((dx * dx + dy * dy) ** 0.5),
                                "bbox_anchor_w_ratio": w_ratio,
                                "bbox_anchor_h_ratio": h_ratio,
                                "bbox_anchor_area_ratio": area_ratio,
                                "bbox_anchor_log_w_ratio": float(np.log(max(w_ratio, 1e-12))),
                                "bbox_anchor_log_h_ratio": float(np.log(max(h_ratio, 1e-12))),
                                "bbox_anchor_log_area_ratio": float(np.log(max(area_ratio, 1e-12))),
                                "bbox_anchor_aspect_ratio_diff": (w / max(h, 1e-12)) - anchor_aspect_ratio,
                                "bbox_anchor_ciou": bbox_anchor_ciou,
                                "bbox_anchor_ciou_loss": 1.0 - bbox_anchor_ciou,
                                "obj": obj,
                                "cls_conf": cls_conf,
                                "score": score,
                                "pred_class": pred_class_names[pred_idx] if pred_idx < len(pred_class_names) else _class_name(detector.names, cls_idx),
                                "pred_class_id": cls_idx,
                                **null_stats,
                                **cand_stats,
                                **gradient_stats,
                                "max_iou": float(best_ious[pred_idx]) if pred_idx < len(best_ious) else 0.0,
                                "tp": int(tp_flags[pred_idx]) if pred_idx < len(tp_flags) else 0,
                            }
                        )
                raw_prof.end(t_raw, batch_items)
                del infer_batch, raw_prediction, raw_logits, raw_priors, selected_preds, selected_indices
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if grad_buffer is not None:
            grad_buffer.remove()

    summary = {
        "output_csv": str(output_csv),
        "total_images": int(total_images),
        "total_predictions": int(total_predictions),
        "mean_predictions_per_image": (float(total_predictions) / total_images) if total_images else 0.0,
        "gradient_enabled": bool(gradient_enabled),
        "gradient_layers": list(gradient_layers),
        "gradient_targets": list(PREDICTION_DUMP_GRAD_TARGETS),
    }
    with open(run_dir / "prediction_dump_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = raw_prof.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved prediction dump summary: {run_dir / 'prediction_dump_summary.json'}")
    print(f"Saved raw compute timing: {timing_csv}")
    print(f"Saved raw compute timing summary: {timing_json}")


__all__ = ["run_prediction_dump_csv"]
